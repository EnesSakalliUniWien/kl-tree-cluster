---
title: Statistic Distribution Shape Panel 2026-06-13
type: source
status: reviewed
updated: 2026-06-13
sources:
  - benchmarks/diagnostics/calibration/statistic_distribution_shape_panel.py
  - tests/validation/93_test_statistic_distribution_shape_panel.py
  - benchmarks/validation/selected_edge_type1_geometry.py
  - benchmarks/diagnostics/calibration/differential_statistic_validity_panel.py
  - kl_clustering_analysis/hierarchy_analysis/statistics/projection/projected_wald/projected_wald_reference_distribution.py
  - kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/pair_testing/wald_statistic/sibling_divergence_test.py
  - raw/assets/benchmark-results/selected-edge-type1-binary-categorical-pilot-20260604/merged/selected_edge_geometry_edges.csv
tags:
  - source
  - diagnostics
  - calibration
  - distribution
---

# Statistic Distribution Shape Panel 2026-06-13

## Summary

`statistic_distribution_shape_panel.py` separates distribution skew that is
expected from chi-square degrees of freedom from residual statistic-shape and
tail misalignment. The diagnostic also compares the current projected-Wald
reference law against an alternate covariance-inferred Satterthwaite reference
when alternate degrees of freedom and scale are available.

## Key Points

- Current projected-Wald sibling tests use a fixed-subspace chi-square law with
  degrees of freedom equal to the projection dimension.
- The diagnostic records the expected chi-square skew implied by the observed
  df mixture, using \(\operatorname{skew}(\chi^2_\nu)=\sqrt{8/\nu}\) and the
  mixture moments for rows with different dfs.
- The alternate covariance-inferred law uses
  \(c\chi^2_{\nu_{\mathrm{eff}}}\), where
  \(\nu_{\mathrm{eff}}=(\sum_i\lambda_i)^2/\sum_i\lambda_i^2\) and
  \(c=\sum_i\lambda_i^2/\sum_i\lambda_i\).
- Stored 2026-06-04 selected-edge sibling rows still cannot be used for this
  sibling statistic panel because raw sibling statistics are missing.
- Future selected-edge geometry exports now emit raw sibling statistic,
  raw chi-square p-value, adjusted sibling statistic/p-value when available,
  covariance-inferred df/scale, and covariance Laplacian summaries.
- On live method-proof sibling rows, covariance-inferred effective df was
  nearly identical to the current projection df, while the inferred covariance
  scale was large. This points away from df count alone and toward covariance
  scale, anisotropy, and selection as the remaining distribution-shape drivers.
- The differential statistic-validity panel adds a pre-tail diagnostic layer:
  if Fisher boundary geometry, whitening, projection, or selection geometry is
  unstable, distribution-shape mismatch should not be repaired by df tuning
  alone.

## Evidence

- `tests/validation/93_test_statistic_distribution_shape_panel.py` verifies
  df-mixture skew accounting, alternate covariance-inferred df/scale
  comparison, invalid-row rejection, and output writing.
- Stored selected-edge edge statistics show severe chi-square tail mismatch
  under current df: edge tail rates near `0.895`--`1.000` at alpha `0.05`
  across fixed-tree and selected-tree binary/categorical groups.
- Live method-proof sibling rows produced `756` raw sibling records with
  covariance-inferred alternate df for every row. Current df means were about
  `1.24`--`1.31`; alternate df means were about `1.24`--`1.30`; alternate
  reference scales were about `56.5`, `56.6`, `88.8`, and `133.1` by case.
- Current sibling chi-square tail rates were about `0.995`--`1.000`; alternate
  covariance-scaled tail rates dropped to about `0.261`--`0.289`, still above
  nominal.
- A one-replicate `binary_2clusters` differential smoke found that
  `wald_metric_boundary_unstable` dominated both fixed-tree and selected-tree
  group summaries, explaining why the chi-square tail problem cannot be treated
  as only a post-selection df issue in those rows.

## Links

- [[selected-edge-sibling-null-equation-20260613]]
- [[production-admissibility-contract-20260613]]
- [[projected-wald-statistic]]
- [[differential-statistic-validity-panel-20260613]]
- [[open-mathematical-questions]]
