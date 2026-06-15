---
title: Covariance Laplacian Panel 2026-06-13
type: source
status: reviewed
updated: 2026-06-13
sources:
  - benchmarks/diagnostics/calibration/covariance_laplacian_panel.py
  - tests/validation/94_test_covariance_laplacian_panel.py
  - benchmarks/validation/selected_edge_type1_geometry.py
  - benchmarks/diagnostics/calibration/differential_statistic_validity_panel.py
  - kl_clustering_analysis/hierarchy_analysis/statistics/contrast_covariance.py
  - kl_clustering_analysis/hierarchy_analysis/statistics/projection/spectral/tree_estimator.py
  - wiki/questions/open-mathematical-questions.md
tags:
  - source
  - diagnostics
  - calibration
  - covariance
  - laplacian
---

# Covariance Laplacian Panel 2026-06-13

## Summary

`covariance_laplacian_panel.py` adds a graph-Laplacian view of covariance
matrices. It converts covariance to absolute correlation, removes the diagonal,
and reports connected components, algebraic connectivity, off-diagonal mass,
effective rank, and Laplacian status.

## Key Points

- The panel distinguishes diagonal covariance, disconnected block covariance,
  near-disconnected covariance, and connected covariance.
- The diagnostic is scale-invariant because graph edges are built from absolute
  correlation rather than raw covariance magnitude.
- The live method-proof check exposed two different covariance objects that
  should not be conflated:
  - sibling contrast covariance for Bernoulli features is diagonal by
    construction, so its Laplacian graph has one isolated component per
    feature;
  - reconstructed parent spectral covariance is dense and mostly connected,
    because it comes from selected descendant tangent covariance and PCA
    geometry.
- This means the previous covariance evidence covered eigenvalue scale and
  selected-subspace summaries but left out graph connectivity and the
  distinction between sibling whitening covariance and parent spectral
  covariance.
- `selected_edge_type1_geometry.py` now exports bounded Laplacian summaries for
  sibling contrast covariance and parent spectral covariance on each sibling
  row, using `laplacian_skipped_high_dimension` when the matrix exceeds the
  diagnostic cap.
- The differential statistic-validity panel uses the parent spectral
  eigenvalues in a complementary way: eigengap and recomputed-projection
  sensitivity diagnose Grassmannian projection instability, while the
  Laplacian panel diagnoses covariance graph connectivity.

## Evidence

- `tests/validation/94_test_covariance_laplacian_panel.py` verifies diagonal,
  disconnected block, connected dense covariance, invalid matrix rejection,
  summary output, and CSV runner behavior.
- On live method-proof sibling contrast covariance matrices, `24/24` top-parent
  matrices were `laplacian_diagonal_covariance`; off-diagonal mass was zero and
  the largest component fraction was `1 / p`.
- On reconstructed parent spectral covariance matrices for top live
  method-proof parents, `22/24` were `laplacian_connected_covariance` and
  `2/24` were disconnected at absolute-correlation threshold `0.05`.
- Parent spectral covariance rows had median off-diagonal absolute mass near
  `0.991`, median largest component fraction `1.0`, and median effective rank
  near `2.0`.

## Wiki Coverage Gaps

- The wiki has strong coverage of MP eigenvalue thresholds, selected-tail
  spectral covariates, and feature-covariance validation, but did not have a
  Laplacian/connectivity view of covariance matrices.
- The wiki did not clearly separate the diagonal Bernoulli sibling contrast
  covariance used for whitening from the dense parent spectral covariance used
  to select projection directions.
- The wiki did not yet state that covariance scale can move the sibling tail
  rates substantially while covariance-inferred effective df remains close to
  current projection df.
- The wiki still lacks production-scale raw sibling statistic exports, so
  stored selected-edge sibling artifacts cannot be reanalyzed with the
  distribution-shape or selected edge+sibling equation panels.

## Links

- [[statistic-distribution-shape-panel-20260613]]
- [[selected-edge-sibling-null-equation-20260613]]
- [[differential-statistic-validity-panel-20260613]]
- [[local-marchenko-pastur-rule]]
- [[open-mathematical-questions]]
