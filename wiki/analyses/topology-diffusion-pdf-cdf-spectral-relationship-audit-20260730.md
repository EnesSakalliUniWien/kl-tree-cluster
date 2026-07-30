---
title: Topology Diffusion PDF CDF Spectral Relationship Audit 20260730
type: analysis
status: draft
updated: 2026-07-30
sources:
  - reports/audits/generated/all_topology_spectral_pdf_cdf_relationships/MATHEMATICAL_RELATIONSHIP_AUDIT.md
  - reports/audits/generated/all_topology_spectral_pdf_cdf_relationships/all_versions_distribution_fit_summary.csv
  - reports/audits/generated/all_topology_spectral_pdf_cdf_relationships/all_versions_pvalue_uniformity_summary.csv
  - tree_break_selection/hierarchy_analysis/statistics/projection/projected_wald/projected_wald_reference_distribution.py
  - tree_break_selection/hierarchy_analysis/statistics/projection/projected_wald/projected_wald_kernel.py
  - tree_break_selection/hierarchy_analysis/statistics/contrast_covariance.py
tags:
  - calibration
  - topology
  - diffusion
  - spectral
---

# Topology Diffusion PDF CDF Spectral Relationship Audit 20260730

## Summary

The all-method PDF/CDF audit shows that sibling projected-Wald statistics do
not follow a plain fixed-subspace chi-square law after topology, PCA basis, and
projection dimension are selected from the same data. Raw statistics are best
approximated by scaled chi-square, exponential, or gamma families across the
tested construction and diffusion variants.

## Details

The implementation computes a whitened sibling contrast, projects it onto the
parent PCA basis, and evaluates the quadratic statistic
`T = ||P_parent z||²` against a fixed-subspace chi-square reference. That
reference is valid only when the projection matrix and dimension are fixed
independently of the tested contrast.

The generated audit fit 18 successful construction/diffusion/branch-source
groups. By KS distance, raw statistic fits split into 9 scaled-chi-square, 6
exponential, and 3 gamma groups. Plain `χ²(df=2)` and the actual-record
`χ²(df)` mixture were not the best raw fit for any group. Diffusion groups were
consistently closer to exponential among the tested families.

The empirical relationship
`log(T / df) ~ log(parent_eigenvalue_sum) + sibling_null_weight + node-size and
spectral-concentration terms` has statistically nonzero spectral coefficients,
but global explanatory power stays below `R² = 0.10`. Parent spectrum is
therefore a real driver but not a complete universal null law.

Branch lengths enter before projection through the contrast covariance variance
multiplier. Fixed-topology NNLS branch lengths can reduce or inflate p-values by
changing this variance scale, but they do not alter topology or parent PCA
eigenvectors.

## Evidence

- `reports/audits/generated/all_topology_spectral_pdf_cdf_relationships/all_versions_pdf_cdf_relationships.pdf`
  contains statistic PDF/CDF and relationship panels for each successful group.
- `reports/audits/generated/all_topology_spectral_pdf_cdf_relationships/all_versions_pvalue_pdf_cdf.pdf`
  contains record and decision p-value PDF/CDF panels against Uniform(0,1).
- `reports/audits/generated/all_topology_spectral_pdf_cdf_relationships/MATHEMATICAL_RELATIONSHIP_AUDIT.md`
  records the implementation equation, distribution-fit counts, p-value
  calibration summary, and current interpretation.
- `tree_break_selection/hierarchy_analysis/statistics/projection/projected_wald/projected_wald_reference_distribution.py`
  states the fixed-subspace chi-square reference and its independence
  limitation.
- `tree_break_selection/hierarchy_analysis/statistics/contrast_covariance.py`
  applies the branch-length variance multiplier before projected-Wald testing.

## Links

- [[tree-construction-method-map]]
- [[adaptive-diffusion-nnls-method-library-audit]]
- [[selected-tail-topology-refinement-20260603]]
- [[projected-wald-statistic]]

## Open Questions

- Whether production calibration should use method/case/node-conditioned
  empirical nulls, a selected-quadratic-form reference, or a narrower validated
  reference family per topology construction method.
- Whether diffusion-specific exponential-like tails are intrinsic to the
  diffusion topology selection step or caused by the current selected sibling
  support rules.
