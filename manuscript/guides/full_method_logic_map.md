# KL-TE Method Logic Map

This guide records the current mathematical contract behind the manuscript.
It is not a results section.

## Central Mathematical Object

KL-TE starts from a discrete sample-feature matrix and a rooted binary
agglomerative hierarchy. Each internal node defines a parent subtree with two
child subtrees. The method computes subtree feature distributions, annotates
child-parent edges and sibling pairs with projected-Wald tests, corrects the
edge and sibling p-values, and walks the tree top down to decide the final
partition.

## Main Assumptions

- The working feature representation is binary or indicator-like.
- Subtree membership is treated as fixed when a local p-value is evaluated.
- Coordinatewise Bernoulli variance formulas are exact only for independent
  Bernoulli coordinates with fixed groups.
- Local PCA directions are treated as fixed once selected.
- The orthonormal projected-Wald reference uses a chi-square law for a fixed
  projected subspace under an isotropic standardized null.
- The empirical sibling inflation model is a calibration model, not an exact
  selective-inference theorem.

## Estimator and Statistic Chain

1. Estimate node-level feature rates from descendant leaves.
2. Form child-parent and sibling contrast vectors.
3. Standardize each coordinate using the corresponding Bernoulli-style
   variance model.
4. Build parent-local PCA directions from the local subtree representation.
5. Select the projection dimension with the local Marchenko-Pastur rule plus
   implementation floor and cap.
6. Compute the projected quadratic statistic in an orthonormal basis.
7. Use a chi-square reference for the uninflated projected-Wald statistic.
8. For sibling tests, estimate context-weighted empirical-null inflation from
   sibling records and divide the raw statistic by that inflation.
9. Correct p-values and traverse the tree.

## Canonical Terminology

- Use **projected-Wald statistic** for the projected quadratic form.
- Use **orthonormal projected-Wald reference** for the chi-square reference
  after projection onto orthonormal rows.
- Use **context-weighted empirical-null inflation** for the sibling calibration
  model.
- Do not call the sibling inflation model an exact selective p-value.
- Do not call the empirical-null weight a posterior null probability.

## Constants and Defaults Requiring Validation

- Edge alpha: `0.001`.
- Sibling alpha: `0.01`.
- Marchenko-Pastur upper-edge dimension rule.
- Minimum spectral dimension: `2`.
- Inclusion of internal spectral rows.
- Sibling projection dimension: geometric mean of child edge dimensions.
- Empirical-null weight from edge-adjusted p-values.
- Context bandwidth over log projection dimension.
- Effective-sample penalty.
- Pass-through traversal.

## Empirical Claims Not Yet Manuscript-Ready

- Null calibration of the edge stage.
- Null calibration of the sibling stage after empirical-null inflation.
- Full-pipeline control of the final number of clusters.
- Power under planted binary subtree structure.
- Robustness to sparse high-dimensional feature matrices.
- Robustness to one-hot categorical dependence.
- Robustness to continuous discretization.
- Real-data interpretability.

## Required Before Submission

[RESULTS GAP: add locked simulation and benchmark outputs with seeds, commit,
configuration manifest, and confidence intervals.]

[VALIDATION GAP: add ablations for every implementation constant listed in the
method-constants table.]

[PROOF GAP: state the exact assumptions under which the projected chi-square
reference is valid after data-dependent local projection, or present it as a
working approximation requiring empirical validation.]
